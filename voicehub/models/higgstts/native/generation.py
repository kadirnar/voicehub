"""Native delayed-codebook generation for Higgs Audio v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from voicehub.generation import create_generator, filter_top_k, filter_top_p
from voicehub.models.higgstts.native.modeling import HiggsAudioV2ForConditionalGeneration
from voicehub.models.higgstts.native.processing import HiggsAudioV2Batch, HiggsAudioV2Processor


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

    @staticmethod
    def _force_token(
        logits: Tensor,
        row_mask: Tensor,
        token_id: int,
    ) -> Tensor:
        if not row_mask.any():
            return logits
        result = logits.clone()
        rows = result.reshape(-1, result.shape[-1])
        flattened_mask = row_mask.reshape(-1)
        selected = rows[flattened_mask, token_id].clone()
        rows[flattened_mask] = -float("inf")
        rows[flattened_mask, token_id] = selected
        return result

    def _decode_completed(
        self,
        delayed: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        if not len(eos_rows):
            raise RuntimeError(
                "Higgs reached `max_new_tokens` before completing the "
                "audio EOS delay pattern.")
        end = int(eos_rows[0, 0])
        delayed_content = after_start[1:end]
        aligned = self.processor.revert_delay_pattern(delayed_content).clamp(
            0,
            self.processor.audio_tokenizer.config.codebook_size - 1,
        )
        audio_codes = aligned.transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            waveform = self.processor.audio_tokenizer.decode(audio_codes).audio_values
        return waveform, audio_codes

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
            constrained = self._force_token(
                next_logits,
                bos_mask,
                config.audio_stream_bos_id,
            )
            constrained = self._force_token(
                constrained,
                eos_mask | finished.unsqueeze(-1),
                config.audio_stream_eos_id,
            )
            next_codes = self._sample(
                constrained,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=request_generator,
            )

            if ras_window is not None and delayed_frames:
                history = torch.stack(delayed_frames, dim=1)
                window = history[:, -ras_window:]
                repeated = window == next_codes.unsqueeze(1)
                repeated &= ((window != config.audio_stream_bos_id) & (window != config.audio_stream_eos_id))
                replacement_mask = (repeated.sum(dim=1) >= ras_max_repeats)
                replacement_mask &= ~(bos_mask | eos_mask)
                if replacement_mask.any():
                    raw_probabilities = torch.softmax(
                        next_logits[replacement_mask].float(),
                        dim=-1,
                    )
                    replacements = torch.multinomial(
                        raw_probabilities,
                        num_samples=1,
                        replacement=True,
                        generator=request_generator,
                    ).squeeze(-1)
                    next_codes = next_codes.clone()
                    next_codes[replacement_mask] = replacements

            delayed_frames.append(next_codes)
            has_eos = (next_codes == config.audio_stream_eos_id).any(dim=-1)
            newly_started = (eos_age < 0) & has_eos
            eos_age[newly_started] = 0
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

            frame = next_codes[:, None]
            frame_mask = (~all_eos)[:, None]
            output = self.model(
                next_text[:, None],
                audio_input_ids=frame,
                audio_input_ids_mask=frame_mask,
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
        waveform, audio_codes = self._decode_completed(delayed)
        return HiggsAudioV2GenerationOutput(
            waveform=waveform,
            audio_codes=audio_codes,
            delayed_audio_codes=delayed,
            text_sequence=input_ids,
            sample_rate=self.processor.sample_rate,
            generated_steps=delayed.shape[1],
        )


__all__ = [
    "HiggsAudioV2GenerationOutput",
    "HiggsAudioV2Generator",
]
