"""Complete VoiceHub-native XTTS v2 graph."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.xtts.native.audio import cloning_mel, load_reference_audio
from voicehub.models.xtts.native.configuration import XTTS2Config
from voicehub.models.xtts.native.decoder import HifiDecoder
from voicehub.models.xtts.native.gpt import XTTS2GPT
from voicehub.optimization.protocols import OptimizationCompileTarget


class XTTS2Model(nn.Module):
    """Checkpoint-compatible XTTS v2 GPT, speaker encoder, and vocoder."""

    def __init__(
        self,
        config: XTTS2Config,
        *,
        start_text_token: int,
        stop_text_token: int,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        args = config.model_args
        self.gpt = XTTS2GPT(
            layers=args.gpt_layers,
            model_dim=args.gpt_n_model_channels,
            start_text_token=start_text_token,
            stop_text_token=stop_text_token,
            heads=args.gpt_n_heads,
            max_text_tokens=args.gpt_max_text_tokens,
            max_mel_tokens=args.gpt_max_audio_tokens,
            max_prompt_tokens=args.gpt_max_prompt_tokens,
            number_text_tokens=args.gpt_number_text_tokens,
            num_audio_tokens=args.gpt_num_audio_tokens,
            start_audio_token=args.gpt_start_audio_token,
            stop_audio_token=args.gpt_stop_audio_token,
            use_perceiver_resampler=args.gpt_use_perceiver_resampler,
            code_stride_len=args.gpt_code_stride_len,
        )
        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=args.input_sample_rate,
            output_sample_rate=args.output_sample_rate,
            output_hop_length=args.output_hop_length,
            ar_mel_length_compression=args.gpt_code_stride_len,
            decoder_input_dim=args.decoder_input_dim,
            d_vector_dim=args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=(args.cond_d_vector_in_each_upsampling_layer),
        )
        self.register_buffer("mel_stats", torch.ones(80))

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose GPT and vocoder boundaries reached by XTTS v2."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "gpt.forward",
                self.gpt,
                "forward",
            ), )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "gpt.forward",
                self.gpt,
                "forward",
            ),
            OptimizationCompileTarget(
                "hifigan_decoder.forward",
                self.hifigan_decoder,
                "forward",
            ),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return self.gpt(*args, **kwargs)

    @torch.inference_mode()
    def conditioning_latents(
        self,
        reference_audio: str | Path | Sequence[str | Path],
        *,
        max_ref_length: int | None = None,
        gpt_cond_length: int | None = None,
        chunk_length: int | None = None,
        sound_norm_refs: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        max_ref_length = (self.config.max_ref_len if max_ref_length is None else max_ref_length)
        gpt_cond_length = (self.config.gpt_cond_len if gpt_cond_length is None else gpt_cond_length)
        chunk_length = (self.config.gpt_cond_chunk_len if chunk_length is None else chunk_length)
        sound_norm_refs = (self.config.sound_norm_refs if sound_norm_refs is None else sound_norm_refs)
        if isinstance(reference_audio, (str, Path)):
            references = (reference_audio, )
        else:
            references = tuple(reference_audio)
        if not references:
            raise ValueError("XTTS v2 requires at least one reference audio file.")
        audios = []
        speakers = []
        model_dtype = next(self.parameters()).dtype
        for reference in references:
            audio = load_reference_audio(
                reference,
                sample_rate=self.config.audio.sample_rate,
                device=self.device,
            ).to(dtype=model_dtype)
            audio = audio[:, :self.config.audio.sample_rate * max_ref_length]
            if sound_norm_refs:
                peak = audio.abs().amax()
                if peak > 0:
                    audio = audio / peak * 0.75
            speaker_audio = audio
            if self.config.audio.sample_rate != 16_000:
                from voicehub.processing.waveform import resample_waveform

                speaker_audio = resample_waveform(
                    audio.squeeze(0),
                    self.config.audio.sample_rate,
                    16_000,
                ).unsqueeze(0)
            speakers.append(
                self.hifigan_decoder.speaker_encoder(
                    speaker_audio,
                    l2_norm=True,
                ).unsqueeze(-1))
            audios.append(audio)
        speaker = torch.stack(speakers).mean(dim=0)
        audio = torch.cat(audios, dim=-1)
        audio = audio[:, :self.config.audio.sample_rate * gpt_cond_length]
        styles = []
        chunk_samples = self.config.audio.sample_rate * chunk_length
        for start in range(0, audio.shape[-1], chunk_samples):
            chunk = audio[:, start:start + chunk_samples]
            if chunk.shape[-1] < self.config.audio.sample_rate // 3:
                continue
            mel = cloning_mel(
                chunk,
                self.mel_stats,
                sample_rate=self.config.audio.sample_rate,
            )
            styles.append(self.gpt.get_style_emb(mel))
        if not styles:
            raise ValueError("XTTS v2 reference audio must be at least 0.33 seconds.")
        conditioning = torch.stack(styles).mean(dim=0).transpose(1, 2)
        return conditioning, speaker

    @torch.inference_mode()
    def synthesize_tokens(
        self,
        text_tokens: Tensor,
        conditioning: Tensor,
        speaker_embedding: Tensor,
        *,
        speed: float = 1.0,
        **generation_options,
    ) -> Tensor:
        codes = self.gpt.generate(
            conditioning,
            text_tokens,
            **generation_options,
        )
        text_lengths = torch.full(
            (text_tokens.shape[0], ),
            text_tokens.shape[1],
            device=text_tokens.device,
            dtype=torch.long,
        )
        wav_lengths = torch.full(
            (codes.shape[0], ),
            max(1, codes.shape[1] - 3) * self.gpt.code_stride_len,
            device=codes.device,
            dtype=torch.long,
        )
        latents = self.gpt(
            text_tokens,
            text_lengths,
            codes,
            wav_lengths,
            cond_latents=conditioning,
            return_latent=True,
        )
        if speed != 1.0:
            latents = F.interpolate(
                latents.transpose(1, 2),
                scale_factor=1.0 / speed,
                mode="linear",
            ).transpose(1, 2)
        return self.hifigan_decoder(latents, g=speaker_embedding)


__all__ = ["XTTS2Model"]
