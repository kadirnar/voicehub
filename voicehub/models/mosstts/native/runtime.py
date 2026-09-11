"""Native MOSS-TTS lifecycle shared by inference and fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.hub import read_json_file
from voicehub.models.mosstts.native.artifacts import MossTTSArtifacts, resolve_mosstts_artifacts
from voicehub.models.mosstts.native.checkpoint import (
    MossCheckpointReport,
    load_mosstts_checkpoint,
    save_mosstts_pretrained,
)
from voicehub.models.mosstts.native.codec import (
    MossAudioCodec,
    MossAudioCodecConfig,
    MossCodecDecodeOutput,
    MossCodecUnavailable,
    NativeMossAudioCodec,
)
from voicehub.models.mosstts.native.configuration import MossTTSConfig
from voicehub.models.mosstts.native.metadata import (
    MOSS_CODEC_V1_REPOSITORY,
    MOSS_CODEC_V2_REPOSITORY,
    MOSS_TTS_CHECKPOINTS,
)
from voicehub.models.mosstts.native.modeling import MossRealtimeModel, MossTTSModel, MossTTSOutput, build_mosstts_model
from voicehub.models.mosstts.native.processing import MossGeneratedCodes, MossProcessorBatch, MossTTSProcessor
from voicehub.models.mosstts.native.tokenization import MossTextTokenizer
from voicehub.processing.waveform import NativeAudio, load_pcm_wave, normalize_waveform, resample_waveform_kaiser


def resolve_mosstts_dtype(
    dtype_name: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError("MOSS-TTS compute dtype must be a non-empty string.")
    aliases = {
        "auto": "bfloat16",
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "float": "float32",
    }
    normalized = aliases.get(
        dtype_name.strip().lower(),
        dtype_name.strip().lower(),
    )
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"Unsupported MOSS-TTS compute dtype {dtype_name!r}.")
    if torch.device(device).type == "cpu" and dtype in {
            torch.float16,
            torch.bfloat16,
    }:
        return torch.float32
    return dtype


def default_mosstts_codec_config(model_config: MossTTSConfig, ) -> MossAudioCodecConfig:
    """Return the audited boundary for the model's separate codec artifact."""
    if model_config.variant == "local_v1_5":
        return MossAudioCodecConfig(
            version=2,
            sample_rate=48_000,
            downsample_rate=3_840,
            channels=2,
            code_dimension=768,
            rvq_dimension=512,
            output_dimension=768,
            num_quantizers=32,
            codebook_size=1_024,
            codebook_dimension=8,
            quantizer_type="rlfq",
            channel_interleave=True,
        )
    return MossAudioCodecConfig(
        version=1,
        sample_rate=24_000,
        downsample_rate=1_920,
        channels=1,
        code_dimension=768,
        rvq_dimension=512,
        output_dimension=768,
        num_quantizers=32,
        codebook_size=1_024,
        codebook_dimension=8,
        quantizer_type="rlfq",
        channel_interleave=False,
    )


class MossTTSRuntime(nn.Module):
    """Compose the trainable MOSS language graph with its native codec."""

    def __init__(
        self,
        *,
        model: MossTTSModel,
        tokenizer: MossTextTokenizer,
        processor: MossTTSProcessor,
        artifacts: MossTTSArtifacts | None = None,
        checkpoint_report: MossCheckpointReport | None = None,
        codec: MossAudioCodec | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module) or not isinstance(
                getattr(model, "config", None),
                MossTTSConfig,
        ):
            raise TypeError("`model` must be a native MOSS-TTS graph.")
        if not isinstance(tokenizer, MossTextTokenizer):
            raise TypeError("`tokenizer` must be MossTextTokenizer.")
        if not isinstance(processor, MossTTSProcessor):
            raise TypeError("`processor` must be MossTTSProcessor.")
        if processor.config != model.config:
            raise ValueError("MOSS-TTS processor and model configurations disagree.")
        if processor.tokenizer is not tokenizer:
            raise ValueError("MOSS-TTS processor must use the supplied tokenizer.")
        codec_config = default_mosstts_codec_config(model.config)
        resolved_codec: MossAudioCodec = (MossCodecUnavailable(codec_config) if codec is None else codec)
        if not isinstance(resolved_codec, MossAudioCodec):
            raise TypeError("`codec` must implement the native MossAudioCodec protocol.")
        if resolved_codec.config.sample_rate != model.config.sample_rate:
            raise ValueError("MOSS-TTS model and codec sample rates disagree.")
        if resolved_codec.config.codebook_size != model.config.audio_vocab_size:
            if model.config.variant != "realtime":
                raise ValueError("MOSS-TTS model and codec codebook sizes disagree.")
            # Realtime reserves three protocol IDs in its 1027-wide audio
            # graph while the separately versioned codec remains 1024-wide.
            if (resolved_codec.config.codebook_size != 1_024 or model.config.audio_vocab_size != 1_027):
                raise ValueError("Realtime MOSS codec vocabulary is incompatible.")
        if resolved_codec.config.num_quantizers < model.config.n_vq:
            raise ValueError("MOSS-TTS codec exposes fewer quantizers than the model.")

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.codec = resolved_codec
        self.artifacts = artifacts
        self.checkpoint_report = checkpoint_report

    @property
    def config(self) -> MossTTSConfig:
        return self.model.config

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def has_waveform_codec(self) -> bool:
        return not isinstance(self.codec, MossCodecUnavailable)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        channelwise_loss_weight: tuple[float, ...] | list[float] | None = None,
    ) -> MossTTSOutput:
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            channelwise_loss_weight=channelwise_loss_weight,
        )

    @torch.inference_mode()
    def encode_reference(
        self,
        waveform: Tensor,
    ) -> Tensor:
        values = torch.as_tensor(waveform)
        channels = self.codec.config.channels
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[0] != channels:
            raise ValueError("MOSS reference waveform must have shape "
                             f"[{channels}, time].")
        if values.shape[1] < 1:
            raise ValueError("MOSS reference waveform cannot be empty.")
        parameter = next(self.model.parameters())
        batch = values.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).unsqueeze(0)
        lengths = torch.tensor(
            [batch.shape[-1]],
            dtype=torch.long,
            device=batch.device,
        )
        encoded = self.codec.encode(
            batch,
            lengths,
            num_quantizers=self.config.n_vq,
        )
        codes = encoded.audio_codes
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise RuntimeError("MOSS codec returned an invalid audio-code batch.")
        # The codec protocol uses [batch, time, quantizers].
        length = int(encoded.audio_code_lengths[0].item())
        return self.processor._codes(
            codes[0, :length],
            name="encoded_reference",
        ).detach()

    def _load_codec_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        require_sampling_rate: bool,
    ) -> Tensor:
        """Load, normalize, resample, and channel-match one codec waveform."""
        source_rate = sampling_rate
        if isinstance(audio, NativeAudio):
            if source_rate is not None and source_rate != audio.sampling_rate:
                raise ValueError("Explicit sampling rate conflicts with NativeAudio metadata.")
            source_rate = audio.sampling_rate
            values = audio.waveform
        elif isinstance(audio, Mapping):
            payload_key = next(
                (key for key in ("array", "waveform", "audio", "input_values") if key in audio),
                None,
            )
            if payload_key is None:
                raise ValueError("MOSS audio mappings require array, waveform, audio, or "
                                 "input_values.")
            mapped_rate = audio.get(
                "sampling_rate",
                audio.get("sample_rate"),
            )
            if (source_rate is not None and mapped_rate is not None and int(source_rate) != int(mapped_rate)):
                raise ValueError("Explicit sampling rate conflicts with audio metadata.")
            source_rate = source_rate if source_rate is not None else mapped_rate
            values = audio[payload_key]
        elif isinstance(audio, (str, Path)):
            values, file_rate = load_pcm_wave(
                audio,
                preserve_channels=True,
            )
            if source_rate is not None and int(source_rate) != file_rate:
                raise ValueError("Explicit sampling rate does not match the WAVE file.")
            source_rate = file_rate
        else:
            values = audio

        if source_rate is None:
            if require_sampling_rate:
                raise ValueError("Raw MOSS training audio requires `sampling_rate` metadata.")
            source_rate = self.codec.config.sample_rate
        if (isinstance(source_rate, bool) or not isinstance(source_rate, int) or source_rate <= 0):
            raise ValueError("MOSS audio sampling rate must be a positive integer.")

        waveform = torch.as_tensor(values)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            if waveform.shape[0] <= 8:
                pass
            elif waveform.shape[1] <= 8:
                waveform = waveform.transpose(0, 1)
            else:
                raise ValueError("MOSS waveform channel dimension must contain at most "
                                 "eight channels.")
        else:
            raise ValueError("MOSS audio must have shape [time] or [channels, time].")
        waveform = torch.stack([normalize_waveform(channel) for channel in waveform])
        if int(source_rate) != self.codec.config.sample_rate:
            waveform = resample_waveform_kaiser(
                waveform,
                int(source_rate),
                self.codec.config.sample_rate,
            )

        target_channels = self.codec.config.channels
        if target_channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.shape[0] == 1:
            waveform = waveform.expand(target_channels, -1).clone()
        elif waveform.shape[0] != target_channels:
            raise ValueError(
                f"MOSS codec requires {target_channels} channel(s); found "
                f"{waveform.shape[0]}.")
        return waveform.contiguous()

    def load_reference_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        """Materialize one reference waveform for the active native codec."""
        return self._load_codec_audio(
            audio,
            sampling_rate=sampling_rate,
            require_sampling_rate=False,
        )

    def _prepare_training_record(
        self,
        record: Mapping[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise TypeError(f"MOSS-TTS training record {index} must be a mapping.")
        output = dict(record)
        raw_keys = [key for key in ("audio", "waveform", "audio_path") if record.get(key) is not None]
        has_codes = record.get("speech_tokens") is not None
        if int(has_codes) + len(raw_keys) != 1:
            raise ValueError(
                f"MOSS-TTS training record {index} requires exactly one of "
                "`speech_tokens`, `audio`, `waveform`, or `audio_path`.")
        if not has_codes:
            waveform = self._load_codec_audio(
                record[raw_keys[0]],
                sampling_rate=record.get(
                    "sampling_rate",
                    record.get("sample_rate"),
                ),
                require_sampling_rate=not isinstance(
                    record[raw_keys[0]],
                    (str, Path, NativeAudio),
                ),
            )
            output["speech_tokens"] = self.encode_reference(waveform)

        raw_reference_keys = [
            key for key in ("reference_audio", "reference_audio_path") if record.get(key) is not None
        ]
        if raw_reference_keys and record.get("reference_codes") is not None:
            raise ValueError(
                f"MOSS-TTS training record {index} cannot combine raw "
                "reference audio with `reference_codes`.")
        if len(raw_reference_keys) > 1:
            raise ValueError(f"MOSS-TTS training record {index} has multiple raw "
                             "reference audio fields.")
        if raw_reference_keys:
            reference = self._load_codec_audio(
                record[raw_reference_keys[0]],
                sampling_rate=record.get(
                    "reference_sampling_rate",
                    record.get("sampling_rate"),
                ),
                require_sampling_rate=not isinstance(
                    record[raw_reference_keys[0]],
                    (str, Path, NativeAudio),
                ),
            )
            output["reference_codes"] = (self.encode_reference(reference), )
        return output

    @torch.inference_mode()
    def generate_codes(
        self,
        text: str,
        *,
        reference_codes: Sequence[Tensor] = (),
        instruction: str | None = None,
        duration_tokens: int | None = None,
        quality: str | None = None,
        sound_event: str | None = None,
        ambient_sound: str | None = None,
        language: str | None = None,
        max_new_tokens: int = 4_096,
        **generation_options: Any,
    ) -> tuple[MossGeneratedCodes, ...]:
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        if isinstance(self.model, MossRealtimeModel):
            prompt = self.processor.build_realtime_generation_prompt(
                text,
                reference_codes=reference_codes,
                device=self.device,
            )
            options = dict(generation_options)
            temperature = float(options.pop(
                "audio_temperature",
                options.pop("temperature", 0.8),
            ))
            top_p = float(options.pop(
                "audio_top_p",
                options.pop("top_p", 0.6),
            ))
            top_k = int(options.pop(
                "audio_top_k",
                options.pop("top_k", 30),
            ))
            repetition_penalty = float(
                options.pop(
                    "audio_repetition_penalty",
                    options.pop("repetition_penalty", 1.1),
                ))
            repetition_window = options.pop("repetition_window", 50)
            # Text sampling controls apply to Delay/Local generators. Realtime
            # consumes caller-provided text tokens deterministically.
            options.pop("text_temperature", None)
            options.pop("text_top_k", None)
            options.pop("text_top_p", None)
            options.pop("use_kv_cache", None)
            if options:
                raise ValueError(
                    "Unsupported Realtime generation options: " + ", ".join(sorted(options)) + ".")
            generated = self.model.generate(
                prompt.input_ids,
                attention_mask=prompt.attention_mask,
                text_ids=prompt.text_ids,
                text_cursor=prompt.text_cursor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
            )[0]
            eos = torch.where(generated[:, 0].eq(1_026))[0]
            if eos.numel():
                generated = generated[:int(eos[0].item())]
            invalid_codes = ((generated < 0) | (generated >= self.codec.config.codebook_size))
            if generated.numel() and bool(invalid_codes.any()):
                raise RuntimeError("Realtime generated a protocol token inside codec audio "
                                   "frames.")
            return (MossGeneratedCodes(
                prompt_audio_frames=0,
                audio_codes=generated.detach(),
            ), )
        prompt = self.processor.build_generation_prompt(
            text,
            reference_codes=reference_codes,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
            device=self.device,
        )
        options = dict(generation_options)
        if self.config.variant == "local_v1_5":
            options["max_new_frames"] = max_new_tokens
        else:
            options["max_new_tokens"] = max_new_tokens
        generated = self.model.generate(
            prompt.input_ids,
            attention_mask=prompt.attention_mask,
            **options,
        )
        return self.processor.decode_generated(generated)

    @torch.inference_mode()
    def decode_codes(self, audio_codes: Tensor) -> MossCodecDecodeOutput:
        codes = self.processor._codes(
            audio_codes,
            name="audio_codes",
        ).to(self.device)
        lengths = torch.tensor(
            [codes.shape[0]],
            dtype=torch.long,
            device=codes.device,
        )
        output = self.codec.decode(codes.unsqueeze(0), lengths)
        if not isinstance(output, MossCodecDecodeOutput):
            raise TypeError("MOSS codec `decode` must return MossCodecDecodeOutput.")
        if output.sample_rate != self.sample_rate:
            raise RuntimeError("MOSS codec returned an unexpected sample rate.")
        if (not isinstance(output.waveform, Tensor) or output.waveform.ndim not in {2, 3} or
                output.waveform.shape[0] != 1):
            raise RuntimeError("MOSS codec returned an invalid waveform batch.")
        return output

    @torch.inference_mode()
    def infer(
        self,
        text: str,
        **kwargs: Any,
    ) -> tuple[MossCodecDecodeOutput, ...]:
        generated = self.generate_codes(text, **kwargs)
        if not generated:
            raise RuntimeError("MOSS-TTS generated no audio codes.")
        return tuple(self.decode_codes(item.audio_codes) for item in generated if item.audio_codes.numel())

    def prepare_training_batch(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> MossProcessorBatch:
        materialized = [
            self._prepare_training_record(record, index=index) for index, record in enumerate(records)
        ]
        return self.processor.collate_training(
            materialized,
            device=self.device,
        )

    def prepare_for_training(self) -> None:
        self.model.train()
        codec = self.codec
        if isinstance(codec, nn.Module):
            codec.eval()
            codec.requires_grad_(False)

    def prepare_for_inference(self) -> None:
        self.model.eval()
        codec = self.codec
        if isinstance(codec, nn.Module):
            codec.eval()

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        state_override: Mapping[str, Tensor] | None = None,
    ) -> Path:
        destination = save_mosstts_pretrained(
            self.model,
            directory,
            state_override=state_override,
        )
        self.tokenizer.save_pretrained(destination)
        return destination


def load_mosstts_runtime(
    source: str | Path,
    *,
    device: str | torch.device = "cpu",
    compute_dtype: str = "bfloat16",
    variant: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    codec: MossAudioCodec | None = None,
    codec_source: str | Path | None = None,
    codec_revision: str | None = None,
    load_codec: bool = True,
    for_training: bool = False,
) -> MossTTSRuntime:
    """Load one native graph after strict immutable artifact validation."""
    artifacts = resolve_mosstts_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = MossTTSConfig.from_dict(
        read_json_file(artifacts.config),
        variant=variant,
    )
    dtype = resolve_mosstts_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = build_mosstts_model(
            config,
            initialize=False,
            dtype=dtype,
        )
    report = load_mosstts_checkpoint(
        model,
        artifacts.checkpoint,
        device=device,
        dtype=dtype,
        source=(artifacts.source if artifacts.source in MOSS_TTS_CHECKPOINTS else None),
        revision=artifacts.revision,
    )
    tokenizer = MossTextTokenizer.from_files(
        artifacts.vocabulary,
        artifacts.merges,
        artifacts.tokenizer_config,
        model_config=config,
    )
    processor = MossTTSProcessor(config, tokenizer)
    if codec is None and load_codec:
        resolved_codec_source = codec_source
        if resolved_codec_source is None:
            resolved_codec_source = (
                MOSS_CODEC_V2_REPOSITORY if config.variant == "local_v1_5" else MOSS_CODEC_V1_REPOSITORY)
        codec = NativeMossAudioCodec.from_pretrained(
            resolved_codec_source,
            revision=codec_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            device=device,
        )
    runtime = MossTTSRuntime(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        artifacts=artifacts,
        checkpoint_report=report,
        codec=codec,
    )
    if for_training:
        runtime.prepare_for_training()
    else:
        runtime.prepare_for_inference()
    return runtime


__all__ = [
    "MossTTSRuntime",
    "default_mosstts_codec_config",
    "load_mosstts_runtime",
    "resolve_mosstts_dtype",
]
