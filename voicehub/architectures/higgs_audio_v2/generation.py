"""Native delayed-codebook generation for Higgs Audio v2."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from voicehub.architectures.higgs_audio_v2.modeling import HiggsAudioV2ForConditionalGeneration
from voicehub.architectures.higgs_audio_v2.processing import HiggsAudioV2Batch, HiggsAudioV2Processor
from voicehub.generation import create_generator, filter_top_k, filter_top_p


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _sampling_options(
    *,
    temperature: Real,
    top_k: int | None,
    top_p: Real,
) -> tuple[float, int | None, float]:
    if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
            not math.isfinite(float(temperature)) or float(temperature) < 0.0):
        raise ValueError("`temperature` must be a finite non-negative number.")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
        raise ValueError("`top_k` must be a positive integer or None.")
    if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(float(top_p)) or
            not 0.0 <= float(top_p) <= 1.0):
        raise ValueError("`top_p` must be finite and in [0, 1].")
    if float(temperature) > 0.0 and float(top_p) == 0.0:
        raise ValueError("Sampling requires `top_p` greater than zero.")
    return float(temperature), top_k, float(top_p)


@dataclass(frozen=True)
class HiggsAudioV2GenerationOutput:
    """Generated delayed/aligned codes and decoded 24 kHz waveform."""

    waveform: Tensor
    audio_codes: Tensor
    delayed_audio_codes: Tensor
    text_sequence: Tensor
    sample_rate: int
    generated_steps: int
    finish_reason: str = "stop"


class HiggsAudioV2Generator:
    """Request-local greedy/sampling loop with official delay semantics."""

    def __init__(
        self,
        model: HiggsAudioV2ForConditionalGeneration,
        processor: HiggsAudioV2Processor,
    ) -> None:
        if not isinstance(model, HiggsAudioV2ForConditionalGeneration):
            raise TypeError("`model` must be HiggsAudioV2ForConditionalGeneration.")
        if not isinstance(processor, HiggsAudioV2Processor):
            raise TypeError("`processor` must be HiggsAudioV2Processor.")
        if model.config != processor.model_config:
            raise ValueError("Higgs generator model and processor configurations differ.")
        self.model = model
        self.processor = processor

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _sample(
        self,
        logits: Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float,
        generator: torch.Generator,
    ) -> Tensor:
        rows = logits.reshape(-1, logits.shape[-1]).float()
        if temperature == 0.0:
            return rows.argmax(dim=-1).reshape(logits.shape[:-1])
        rows = rows / temperature
        if top_k is not None:
            rows = filter_top_k(rows, top_k)
        if top_p < 1.0:
            rows = filter_top_p(rows, top_p)
        probabilities = torch.softmax(rows, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Higgs sampling produced non-finite probabilities.")
        selected = torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        return selected.reshape(logits.shape[:-1])

    def _sample_frame(
        self,
        logits: Tensor,
        history: Tensor | None,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float,
        ras_window: int | None,
        ras_max_repeats: int,
        generator: torch.Generator,
    ) -> Tensor:
        """Sample one delayed frame before delay-pattern constraints.

        Like the source, every codebook is sampled from the processed
        logits and repetition-aware sampling compares it with the most
        recent ``ras_window`` frames of the whole audio stream
        (reference context, BOS/EOS frames included). Repeated codes are
        redrawn from the raw distribution. The caller then imposes the
        BOS/EOS delay pattern, so seeded draws follow the reference
        sampler.
        """
        next_codes = self._sample(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        if ras_window is None or history is None:
            return next_codes
        window = history[:, -ras_window:]
        replacement_mask = ((window == next_codes.unsqueeze(1)).sum(dim=1) >= ras_max_repeats)
        if not replacement_mask.any():
            return next_codes
        replacements = torch.multinomial(
            torch.softmax(logits[replacement_mask].float(), dim=-1),
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        next_codes = next_codes.clone()
        next_codes[replacement_mask] = replacements
        return next_codes

    def _decode_completed(
        self,
        delayed: Tensor,
    ) -> tuple[Tensor, Tensor, str]:
        # Public generation currently accepts one request at a time. Keeping
        # this boundary explicit avoids ragged-code padding entering a codec.
        if delayed.shape[0] != 1:
            raise ValueError("Native Higgs waveform decoding currently supports "
                             "batch_size=1.")
        stream = delayed[0]
        bos = self.model.config.audio_stream_bos_id
        eos = self.model.config.audio_stream_eos_id
        bos_rows = (stream == bos).all(dim=-1).nonzero()
        if not len(bos_rows):
            raise RuntimeError("Higgs generation did not produce its delayed BOS frame.")
        start = int(bos_rows[-1, 0])
        after_start = stream[start:]
        eos_rows = (after_start == eos).all(dim=-1).nonzero()
        truncated = not len(eos_rows)
        if truncated:
            # ``max_new_tokens`` cut the stream before its all-EOS frame.
            # Like the source, return the audio generated so far: keep every
            # aligned frame whose delayed codebooks were all generated.
            end = after_start.shape[0]
            if end - 1 < self.model.config.num_codebooks:
                raise RuntimeError(
                    "Higgs reached `max_new_tokens` before completing one "
                    "audio frame; increase `max_new_tokens`.")
        else:
            end = int(eos_rows[0, 0])
        delayed_content = after_start[1:end]
        aligned = self.processor.revert_delay_pattern(delayed_content)
        # The aligned frame holding the first stream EOS (codebook 0 in the
        # usual case) is not audio. The source serving path drops it
        # (``revert_delay_pattern(...)[:, 1:-1]``) instead of decoding the
        # clamped EOS id as a real code.
        eos_columns = (aligned == eos).any(dim=-1).nonzero()
        if len(eos_columns):
            aligned = aligned[:int(eos_columns[0, 0])]
            truncated = False
        if not len(aligned):
            raise RuntimeError("Higgs generation produced no complete audio frame.")
        aligned = aligned.clamp(
            0,
            self.processor.audio_tokenizer.config.codebook_size - 1,
        )
        audio_codes = aligned.transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            waveform = self.processor.audio_tokenizer.decode(audio_codes).audio_values
        return waveform, audio_codes, "length" if truncated else "stop"

    @torch.no_grad()
    def generate(
        self,
        batch: HiggsAudioV2Batch,
        *,
        max_new_tokens: int = 1_024,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float = 0.95,
        ras_window: int | None = 7,
        ras_max_repeats: int = 2,
        seed: int | None = None,
    ) -> HiggsAudioV2GenerationOutput:
        max_new_tokens = _positive_integer(
            "max_new_tokens",
            max_new_tokens,
        )
        temperature, top_k, top_p = _sampling_options(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if ras_window is not None:
            ras_window = _positive_integer("ras_window", ras_window)
        ras_max_repeats = _positive_integer(
            "ras_max_repeats",
            ras_max_repeats,
        )
        if not isinstance(batch, HiggsAudioV2Batch):
            raise TypeError("`batch` must be HiggsAudioV2Batch.")
        if batch.input_ids.shape[0] != 1:
            raise ValueError("Native Higgs generation currently accepts one prompt.")
        if batch.labels is not None or batch.audio_labels is not None:
            raise ValueError("Generation batches cannot contain labels.")
        device = self.device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        reference_codes = (None if batch.audio_input_ids is None else batch.audio_input_ids.to(device))
        reference_mask = (
            None if batch.audio_input_ids_mask is None else batch.audio_input_ids_mask.to(device))
        self.model.eval()
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            audio_input_ids=reference_codes,
            audio_input_ids_mask=reference_mask,
            use_cache=True,
        )
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Higgs prefill did not create a KV cache.")
        request_generator = create_generator(device, seed)
        config = self.model.config
        codebook_indices = torch.arange(
            config.num_codebooks,
            device=device,
        ).unsqueeze(0)
        eos_age = torch.full(
            (1, ),
            -1,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(1, dtype=torch.bool, device=device)
        delayed_frames = []
        # Repetition-aware sampling looks back over the whole audio stream,
        # including the delayed reference context, as in the source.
        history_prefix = []
        if reference_codes is not None:
            valid_reference = (
                torch.ones(reference_codes.shape[:2], dtype=torch.bool, device=device)
                if reference_mask is None else reference_mask.to(dtype=torch.bool))
            history_prefix.append(reference_codes[valid_reference].unsqueeze(0))
        next_logits = output.logits[:, -1].reshape(
            1,
            config.num_codebooks,
            config.codebook_size,
        ).float()

        for step in range(max_new_tokens):
            active_eos = eos_age >= 0
            eos_age[active_eos] += 1
            bos_mask = codebook_indices >= step
            eos_mask = (active_eos.unsqueeze(-1) & (codebook_indices < eos_age.unsqueeze(-1)))
            if step == 0:
                # The source opens each audio segment with a fixed all-BOS
                # frame and draws no random numbers for it.
                next_codes = torch.full_like(
                    codebook_indices,
                    config.audio_stream_bos_id,
                )
            else:
                history = None
                if ras_window is not None:
                    history = torch.cat(
                        [
                            *(prefix[:, -ras_window:] for prefix in history_prefix),
                            torch.stack(delayed_frames[-ras_window:], dim=1),
                        ],
                        dim=1,
                    )
                next_codes = self._sample_frame(
                    next_logits,
                    history,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    ras_window=ras_window,
                    ras_max_repeats=ras_max_repeats,
                    generator=request_generator,
                )
                next_codes = torch.where(
                    bos_mask,
                    config.audio_stream_bos_id,
                    next_codes,
                )
                next_codes = torch.where(
                    eos_mask | finished.unsqueeze(-1),
                    config.audio_stream_eos_id,
                    next_codes,
                )
            has_eos = (next_codes == config.audio_stream_eos_id).any(dim=-1)
            newly_started = (eos_age < 0) & has_eos
            if newly_started.any():
                # Source delay semantics: when codebook k first emits the
                # stream EOS, codebooks below k end in the same frame and the
                # EOS diagonal continues from k + 1 on the next frame.
                first_eos = (next_codes == config.audio_stream_eos_id).int().argmax(dim=-1)
                next_codes = torch.where(
                    newly_started.unsqueeze(-1) & (codebook_indices < first_eos.unsqueeze(-1)),
                    config.audio_stream_eos_id,
                    next_codes,
                )
                eos_age = torch.where(newly_started, first_eos, eos_age)
            delayed_frames.append(next_codes)
            all_eos = (next_codes == config.audio_stream_eos_id).all(dim=-1)
            finished |= all_eos

            next_text = torch.full(
                (1, ),
                config.audio_token_id,
                dtype=torch.long,
                device=device,
            )
            next_text[eos_age >= 0] = config.audio_delay_token_id
            next_text[finished] = config.eos_token_id
            input_ids = torch.cat(
                (input_ids, next_text[:, None]),
                dim=-1,
            )
            if finished.all():
                break

            # Every decode step feeds one audio frame: the step's text slot is
            # an audio placeholder whose embedding is replaced by the frame
            # (a finished stream has already left the loop). Feeding the frame
            # alone selects the audio norms/MLPs without per-layer
            # device-to-host syncs on a text/audio token mask.
            output = self.model(
                audio_input_ids=next_codes[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError("Higgs incremental decoding discarded its cache.")
            next_logits = output.logits[:, -1].reshape(
                1,
                config.num_codebooks,
                config.codebook_size,
            ).float()

        delayed = torch.stack(delayed_frames, dim=1)
        waveform, audio_codes, finish_reason = self._decode_completed(delayed)
        if finish_reason == "length":
            warnings.warn(
                f"Higgs reached `max_new_tokens` ({max_new_tokens}) before "
                "the audio stream ended; returning truncated audio.",
                stacklevel=2,
            )
        return HiggsAudioV2GenerationOutput(
            waveform=waveform,
            audio_codes=audio_codes,
            delayed_audio_codes=delayed,
            text_sequence=input_ids,
            sample_rate=self.processor.sample_rate,
            generated_steps=delayed.shape[1],
            finish_reason=finish_reason,
        )


__all__ = [
    "HiggsAudioV2GenerationOutput",
    "HiggsAudioV2Generator",
]
