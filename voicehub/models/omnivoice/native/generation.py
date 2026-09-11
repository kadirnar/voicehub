"""Native iterative masked-token generation for OmniVoice."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.models.omnivoice.native.codec import HiggsAudioV2Tokenizer
from voicehub.models.omnivoice.native.duration import RuleDurationEstimator
from voicehub.models.omnivoice.native.modeling import OmniVoiceModel
from voicehub.models.omnivoice.native.processing import (
    TEXT_END,
    TEXT_START,
    OmniVoiceTokenizer,
    combine_text,
    style_prompt,
)
from voicehub.processing.waveform import resample_waveform_kaiser

_PROMPT_FORMAT = "voicehub-native-omnivoice-prompt-v1"
_SPLIT_PUNCTUATION = frozenset(".,;:!?。，；：！？")
_CLOSING_MARKS = frozenset("\"'“”‘’）]》>」】")
_ABBREVIATIONS = frozenset({
    "Apr.",
    "Aug.",
    "Capt.",
    "Cmdr.",
    "Col.",
    "Co.",
    "Corp.",
    "Cpl.",
    "Dec.",
    "Dept.",
    "Dr.",
    "Est.",
    "Etc.",
    "Feb.",
    "Ft.",
    "Gen.",
    "Gov.",
    "Hon.",
    "Inc.",
    "Jan.",
    "Jr.",
    "Jul.",
    "Jun.",
    "Ltd.",
    "Lt.",
    "Maj.",
    "Mar.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Mt.",
    "No.",
    "Nov.",
    "Oct.",
    "Prof.",
    "Rep.",
    "Rev.",
    "Sen.",
    "Sep.",
    "Sept.",
    "Sgt.",
    "Sr.",
    "St.",
    "Vs.",
    "approx.",
    "def.",
    "e.g.",
    "fig.",
    "i.e.",
    "vs.",
})


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"`{name}` must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class OmniVoiceGenerationConfig:
    """Published OmniVoice decoding defaults."""

    num_steps: int = 32
    guidance_scale: float = 2.0
    time_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1

    def __post_init__(self) -> None:
        if (isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps <= 0):
            raise ValueError("`num_steps` must be a positive integer.")
        for name in (
                "guidance_scale",
                "layer_penalty_factor",
                "position_temperature",
                "class_temperature",
                "audio_chunk_duration",
                "audio_chunk_threshold",
                "pad_duration",
                "fade_duration",
        ):
            value = _finite_number(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"`{name}` must be non-negative.")
            object.__setattr__(self, name, value)
        time_shift = _finite_number(self.time_shift, name="time_shift")
        if time_shift <= 0:
            raise ValueError("`time_shift` must be greater than zero.")
        object.__setattr__(self, "time_shift", time_shift)
        for name in ("denoise", "postprocess_output"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")


@dataclass(frozen=True, slots=True)
class OmniVoicePrompt:
    """Reusable voice-cloning prompt with a pickle-free artifact format."""

    audio_tokens: Tensor
    reference_text: str
    reference_rms: float

    def __post_init__(self) -> None:
        if (not isinstance(self.audio_tokens, Tensor) or self.audio_tokens.ndim != 2 or
                self.audio_tokens.shape[0] == 0 or self.audio_tokens.shape[1] == 0):
            raise ValueError("Prompt audio tokens must have shape [codebook, frame].")
        if (self.audio_tokens.dtype == torch.bool or self.audio_tokens.is_floating_point() or
                self.audio_tokens.is_complex()):
            raise TypeError("Prompt audio tokens must use an integer dtype.")
        if not isinstance(self.reference_text, str) or not self.reference_text:
            raise ValueError("Prompt reference text must be non-empty.")
        rms = _finite_number(self.reference_rms, name="reference_rms")
        if rms < 0:
            raise ValueError("`reference_rms` must be non-negative.")
        object.__setattr__(self, "reference_rms", rms)

    def save(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser().resolve()
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError("OmniVoice prompt destination must be absent or empty.")
        destination.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            {"audio_tokens": self.audio_tokens.detach().cpu().long()},
            destination / "prompt.safetensors",
            metadata={"format": _PROMPT_FORMAT},
        )
        document = {
            "format": _PROMPT_FORMAT,
            "reference_rms": self.reference_rms,
            "reference_text": self.reference_text,
        }
        (destination / "prompt.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> OmniVoicePrompt:
        source = Path(directory).expanduser().resolve()
        metadata_path = source / "prompt.json"
        tensor_path = source / "prompt.safetensors"
        if not metadata_path.is_file() or not tensor_path.is_file():
            raise FileNotFoundError("OmniVoice prompt requires prompt.json and prompt.safetensors.")
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("format") != _PROMPT_FORMAT:
            raise ValueError("Unsupported OmniVoice prompt format.")
        with SafeTensorReader(tensor_path) as reader:
            if set(reader.keys()) != {"audio_tokens"}:
                raise ValueError("OmniVoice prompt contains an unexpected tensor inventory.")
            tokens = reader.get_tensor("audio_tokens")
        return cls(
            audio_tokens=tokens,
            reference_text=document.get("reference_text"),
            reference_rms=document.get("reference_rms"),
        )


def _mono_waveform(waveform: Tensor) -> Tensor:
    if not isinstance(waveform, Tensor):
        raise TypeError("Reference waveform must be a PyTorch tensor.")
    values = waveform.float()
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError("Reference waveform must be [sample] or [channel, sample].")
    if values.shape[0] <= 8:
        return values.mean(dim=0)
    if values.shape[1] <= 8:
        return values.mean(dim=1)
    raise ValueError("Reference waveform has no plausible channel axis.")


def _gumbel(
    values: Tensor,
    temperature: float,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    uniform = torch.rand(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )
    noise = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)
    return values / temperature + noise


def _top_k(values: Tensor, ratio: float = 0.1) -> Tensor:
    count = math.ceil(ratio * values.shape[-1])
    selected, indices = values.topk(count, dim=-1)
    filtered = torch.full_like(values, -torch.inf)
    filtered.scatter_(-1, indices, selected)
    return filtered


def _time_steps(config: OmniVoiceGenerationConfig, device: torch.device) -> Tensor:
    steps = torch.linspace(
        0.0,
        1.0,
        config.num_steps + 1,
        device=device,
    )
    return config.time_shift * steps / (1.0 + (config.time_shift - 1.0) * steps)


def _chunk_text(text: str, *, maximum_characters: int) -> list[str]:
    """Split on source-compatible punctuation, retaining closing marks."""
    if maximum_characters <= 0:
        raise ValueError("`maximum_characters` must be positive.")
    sentences: list[list[str]] = []
    current: list[str] = []
    for character in text:
        if (not current and sentences and character in _SPLIT_PUNCTUATION | _CLOSING_MARKS):
            sentences[-1].append(character)
            continue
        current.append(character)
        if character not in _SPLIT_PUNCTUATION:
            continue
        abbreviation = False
        if character == ".":
            stripped = "".join(current).strip()
            abbreviation = bool(stripped and stripped.split()[-1] in _ABBREVIATIONS)
        if not abbreviation:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    chunks: list[list[str]] = []
    current = []
    for sentence in sentences:
        if current and len(current) + len(sentence) > maximum_characters:
            chunks.append(current)
            current = []
        current.extend(sentence)
        while len(current) > maximum_characters:
            chunks.append(current[:maximum_characters])
            current = current[maximum_characters:]
    if current:
        chunks.append(current)
    result = ["".join(chunk).strip() for chunk in chunks]
    result = [chunk for chunk in result if chunk]
    if len(result) > 1 and len(result[0]) < 3:
        result[1] = result[0] + result[1]
        result.pop(0)
    if len(result) > 1 and len(result[-1]) < 3:
        result[-2] += result[-1]
        result.pop()
    return result or [text]


class OmniVoiceGenerator:
    """Own text preparation, voice prompting, decoding, and postprocessing."""

    def __init__(
        self,
        model: OmniVoiceModel,
        text_tokenizer: OmniVoiceTokenizer,
        audio_tokenizer: HiggsAudioV2Tokenizer,
    ) -> None:
        if not isinstance(model, OmniVoiceModel):
            raise TypeError("`model` must be a native OmniVoiceModel.")
        if not isinstance(text_tokenizer, OmniVoiceTokenizer):
            raise TypeError("`text_tokenizer` must be an OmniVoiceTokenizer.")
        if not isinstance(audio_tokenizer, HiggsAudioV2Tokenizer):
            raise TypeError("`audio_tokenizer` must be a native HiggsAudioV2Tokenizer.")
        if (model.config.num_audio_codebook != audio_tokenizer.config.num_quantizers):
            raise ValueError("OmniVoice and Higgs codebook counts do not match.")
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.duration_estimator = RuleDurationEstimator()

    @property
    def device(self) -> torch.device:
        return self.model.device

    def create_prompt(
        self,
        waveform: Tensor,
        *,
        sampling_rate: int,
        reference_text: str,
        preprocess_prompt: bool = True,
    ) -> OmniVoicePrompt:
        """Encode a prompt; transcripts are explicit to avoid hidden ASR."""
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError(
                "Native OmniVoice voice cloning requires `reference_text`; "
                "automatic transcription is intentionally outside this model.")
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0):
            raise ValueError("`sampling_rate` must be a positive integer.")
        if not isinstance(preprocess_prompt, bool):
            raise TypeError("`preprocess_prompt` must be a boolean.")
        values = _mono_waveform(waveform)
        if sampling_rate != self.audio_tokenizer.sample_rate:
            values = resample_waveform_kaiser(
                values,
                sampling_rate,
                self.audio_tokenizer.sample_rate,
            )
        if not torch.isfinite(values).all() or values.numel() == 0:
            raise ValueError("Reference waveform is empty or non-finite.")
        rms = float(values.square().mean().sqrt().item())
        codec_input = values
        if 0 < rms < 0.1:
            codec_input = codec_input * (0.1 / rms)
        remainder = codec_input.numel() % self.audio_tokenizer.config.hop_length
        if remainder:
            codec_input = codec_input[:-remainder]
        if codec_input.numel() == 0:
            raise ValueError("Reference audio is shorter than one codec frame.")
        with torch.no_grad():
            codes = self.audio_tokenizer.encode(codec_input.to(
                self.audio_tokenizer.device)[None, None, :]).audio_codes[0].cpu()
        text = reference_text.strip()
        if preprocess_prompt and text[-1] not in ".!?。！？":
            text += "。" if any("\u4e00" <= char <= "\u9fff" for char in text) else "."
        return OmniVoicePrompt(codes, text, rms)

    def estimate_target_frames(
        self,
        text: str,
        *,
        prompt: OmniVoicePrompt | None,
        duration: float | None,
        speed: float,
    ) -> int:
        if duration is not None:
            seconds = _finite_number(duration, name="duration")
            if seconds <= 0:
                raise ValueError("`duration` must be greater than zero.")
            return max(1, int(seconds * self.audio_tokenizer.frame_rate))
        speed_value = _finite_number(speed, name="speed")
        if speed_value <= 0:
            raise ValueError("`speed` must be greater than zero.")
        reference_text = (prompt.reference_text if prompt is not None else "Nice to meet you.")
        reference_frames = (prompt.audio_tokens.shape[-1] if prompt is not None else 25)
        estimate = self.duration_estimator.estimate_duration(
            text,
            reference_text,
            float(reference_frames),
        )
        return max(1, int(estimate / speed_value))

    def _prepare_inputs(
        self,
        text: str,
        *,
        target_frames: int,
        prompt: OmniVoicePrompt | None,
        language: str | None,
        instruction: str | None,
        denoise: bool,
    ) -> tuple[Tensor, Tensor]:
        style_ids = self.text_tokenizer.encode(
            style_prompt(
                language=language,
                instruction=instruction,
                denoise=denoise and prompt is not None,
            )).input_ids
        combined = combine_text(
            text,
            None if prompt is None else prompt.reference_text,
        )
        text_ids = self.text_tokenizer.encode_nonverbal_text(f"{TEXT_START}{combined}{TEXT_END}")
        codebooks = self.model.config.num_audio_codebook
        style = torch.tensor(style_ids, device=self.device).repeat(codebooks, 1)
        text_tokens = torch.tensor(text_ids, device=self.device)
        text_tokens = text_tokens.repeat(codebooks, 1)
        target = torch.full(
            (codebooks, target_frames),
            self.model.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )
        parts = [style, text_tokens]
        if prompt is not None:
            parts.append(prompt.audio_tokens.to(self.device))
        parts.append(target)
        input_ids = torch.cat(parts, dim=1).unsqueeze(0)
        audio_start = style.shape[1] + text_tokens.shape[1]
        audio_mask = torch.zeros(
            (1, input_ids.shape[-1]),
            dtype=torch.bool,
            device=self.device,
        )
        audio_mask[:, audio_start:] = True
        return input_ids, audio_mask

    @torch.no_grad()
    def generate_tokens(
        self,
        text: str,
        *,
        prompt: OmniVoicePrompt | None = None,
        language: str | None = None,
        instruction: str | None = None,
        duration: float | None = None,
        speed: float = 1.0,
        config: OmniVoiceGenerationConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OmniVoice generation text must be non-empty.")
        if prompt is not None:
            if not isinstance(prompt, OmniVoicePrompt):
                raise TypeError("`prompt` must be an OmniVoicePrompt or None.")
            if prompt.audio_tokens.shape[0] != self.model.config.num_audio_codebook:
                raise ValueError("Prompt codebook count does not match OmniVoice.")
        generation = config or OmniVoiceGenerationConfig()
        target_frames = self.estimate_target_frames(
            text,
            prompt=prompt,
            duration=duration,
            speed=speed,
        )
        conditional_ids, conditional_audio_mask = self._prepare_inputs(
            text.strip(),
            target_frames=target_frames,
            prompt=prompt,
            language=language,
            instruction=instruction,
            denoise=generation.denoise,
        )
        conditional_length = conditional_ids.shape[-1]
        unconditional_ids = conditional_ids[..., -target_frames:].clone()
        unconditional_audio_mask = conditional_audio_mask[..., -target_frames:]
        maximum = conditional_length
        input_ids = torch.full(
            (2, self.model.config.num_audio_codebook, maximum),
            self.model.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )
        input_ids[0, :, :conditional_length] = conditional_ids[0]
        input_ids[1, :, :target_frames] = unconditional_ids[0]
        audio_mask = torch.zeros(
            (2, maximum),
            dtype=torch.bool,
            device=self.device,
        )
        audio_mask[0, :conditional_length] = conditional_audio_mask[0]
        audio_mask[1, :target_frames] = unconditional_audio_mask[0]
        attention_mask = torch.zeros(
            (2, 1, maximum, maximum),
            dtype=torch.bool,
            device=self.device,
        )
        attention_mask[0, :, :conditional_length, :conditional_length] = True
        attention_mask[1, :, :target_frames, :target_frames] = True
        if maximum > target_frames:
            padding = torch.arange(target_frames, maximum, device=self.device)
            attention_mask[1, :, padding, padding] = True

        result = torch.full(
            (1, self.model.config.num_audio_codebook, target_frames),
            self.model.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )
        total = result.numel()
        remaining = total
        schedule: list[int] = []
        times = _time_steps(generation, self.device)
        for step in range(generation.num_steps):
            reveal = (
                remaining if step == generation.num_steps - 1 else min(
                    math.ceil(total * float((times[step + 1] - times[step]).item())),
                    remaining,
                ))
            schedule.append(reveal)
            remaining -= reveal
        layer_ids = torch.arange(
            self.model.config.num_audio_codebook,
            device=self.device,
        ).view(1, -1, 1)
        self.model.eval()
        for reveal in schedule:
            if reveal <= 0:
                continue
            logits = self.model(
                input_ids,
                audio_mask,
                attention_mask=attention_mask,
            ).logits.float()
            conditional = logits[
                :1,
                :,
                conditional_length - target_frames:conditional_length,
            ]
            unconditional = logits[1:2, :, :target_frames]
            conditional_log_probs = functional.log_softmax(
                conditional,
                dim=-1,
            )
            if generation.guidance_scale:
                unconditional_log_probs = functional.log_softmax(
                    unconditional,
                    dim=-1,
                )
                log_probs = functional.log_softmax(
                    conditional_log_probs + generation.guidance_scale *
                    (conditional_log_probs - unconditional_log_probs),
                    dim=-1,
                )
            else:
                log_probs = conditional_log_probs
            log_probs[..., self.model.config.audio_mask_id] = -torch.inf
            if generation.class_temperature:
                predicted = _gumbel(
                    _top_k(log_probs),
                    generation.class_temperature,
                    generator=generator,
                ).argmax(dim=-1)
            else:
                predicted = log_probs.argmax(dim=-1)
            scores = log_probs.max(dim=-1).values
            scores -= layer_ids * generation.layer_penalty_factor
            if generation.position_temperature:
                scores = _gumbel(
                    scores,
                    generation.position_temperature,
                    generator=generator,
                )
            scores.masked_fill_(
                result != self.model.config.audio_mask_id,
                -torch.inf,
            )
            unresolved = int((result == self.model.config.audio_mask_id).sum().item())
            indices = scores.flatten().topk(min(reveal, unresolved)).indices
            flat = result.flatten()
            flat[indices] = predicted.flatten()[indices]
            result = flat.view_as(result)
            input_ids[
                :1,
                :,
                conditional_length - target_frames:conditional_length,
            ] = result
            input_ids[1:2, :, :target_frames] = result
        if (result == self.model.config.audio_mask_id).any():
            raise RuntimeError("OmniVoice iterative decoding left masked tokens.")
        return result[0]

    @torch.no_grad()
    def _decode_raw(
        self,
        audio_tokens: Tensor,
    ) -> Tensor:
        return self.audio_tokenizer.decode(audio_tokens.to(
            self.audio_tokenizer.device).unsqueeze(0)).audio_values[0, 0].float()

    def _postprocess(
        self,
        waveform: Tensor,
        *,
        prompt: OmniVoicePrompt | None = None,
        config: OmniVoiceGenerationConfig | None = None,
    ) -> Tensor:
        generation = config or OmniVoiceGenerationConfig()
        if prompt is not None and prompt.reference_rms < 0.1:
            waveform = waveform * (prompt.reference_rms / 0.1)
        elif prompt is None and generation.postprocess_output:
            peak = waveform.abs().max()
            if peak > 1e-6:
                waveform = waveform / peak * 0.5
        padding = int(generation.pad_duration * self.audio_tokenizer.sample_rate)
        fade = min(
            int(generation.fade_duration * self.audio_tokenizer.sample_rate),
            waveform.numel() // 2,
        )
        if fade:
            curve = torch.linspace(
                0.0,
                1.0,
                fade,
                dtype=waveform.dtype,
                device=waveform.device,
            )
            waveform[:fade] *= curve
            waveform[-fade:] *= curve.flip(0)
        if padding:
            waveform = functional.pad(waveform, (padding, padding))
        return waveform

    @torch.no_grad()
    def decode(
        self,
        audio_tokens: Tensor,
        *,
        prompt: OmniVoicePrompt | None = None,
        config: OmniVoiceGenerationConfig | None = None,
    ) -> Tensor:
        return self._postprocess(
            self._decode_raw(audio_tokens),
            prompt=prompt,
            config=config,
        )

    def _merge_chunks(
        self,
        waveforms: list[Tensor],
        *,
        silence_duration: float = 0.3,
    ) -> Tensor:
        if not waveforms:
            raise ValueError("Cannot merge an empty OmniVoice chunk list.")
        if len(waveforms) == 1:
            return waveforms[0]
        total = int(silence_duration * self.audio_tokenizer.sample_rate)
        fade_length = total // 3
        silence_length = fade_length
        merged = waveforms[0].clone()
        for waveform in waveforms[1:]:
            outgoing = min(fade_length, merged.numel())
            if outgoing:
                merged[-outgoing:] *= torch.linspace(
                    1.0,
                    0.0,
                    outgoing,
                    dtype=merged.dtype,
                    device=merged.device,
                )
            incoming_waveform = waveform.clone()
            incoming = min(fade_length, incoming_waveform.numel())
            if incoming:
                incoming_waveform[:incoming] *= torch.linspace(
                    0.0,
                    1.0,
                    incoming,
                    dtype=incoming_waveform.dtype,
                    device=incoming_waveform.device,
                )
            merged = torch.cat([
                merged,
                merged.new_zeros(silence_length),
                incoming_waveform,
            ])
        return merged

    @torch.no_grad()
    def generate(
        self,
        text: str,
        *,
        prompt: OmniVoicePrompt | None = None,
        language: str | None = None,
        instruction: str | None = None,
        duration: float | None = None,
        speed: float = 1.0,
        config: OmniVoiceGenerationConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate one utterance, chunking long text at punctuation."""
        generation = config or OmniVoiceGenerationConfig()
        target_frames = self.estimate_target_frames(
            text,
            prompt=prompt,
            duration=duration,
            speed=speed,
        )
        threshold = int(generation.audio_chunk_threshold * self.audio_tokenizer.frame_rate)
        if (generation.audio_chunk_duration <= 0 or target_frames <= threshold):
            tokens = self.generate_tokens(
                text,
                prompt=prompt,
                language=language,
                instruction=instruction,
                duration=duration,
                speed=speed,
                config=generation,
                generator=generator,
            )
            return self.decode(tokens, prompt=prompt, config=generation)

        average_frames = target_frames / max(1, len(text))
        maximum_characters = max(
            1,
            int(generation.audio_chunk_duration * self.audio_tokenizer.frame_rate / average_frames),
        )
        chunks = _chunk_text(
            text,
            maximum_characters=maximum_characters,
        )
        chunk_speed = speed
        if duration is not None:
            unconstrained_frames = self.estimate_target_frames(
                text,
                prompt=prompt,
                duration=None,
                speed=1.0,
            )
            chunk_speed = unconstrained_frames / target_frames

        token_chunks: list[Tensor] = []
        reference_prompt = prompt
        first_generated_prompt = None
        for index, chunk in enumerate(chunks):
            if index and prompt is None:
                reference_prompt = first_generated_prompt
            tokens = self.generate_tokens(
                chunk,
                prompt=reference_prompt,
                language=language,
                instruction=instruction,
                speed=chunk_speed,
                config=generation,
                generator=generator,
            )
            token_chunks.append(tokens)
            if index == 0 and prompt is None:
                first_generated_prompt = OmniVoicePrompt(
                    tokens.detach().cpu(),
                    chunk,
                    0.1,
                )
        waveform = self._merge_chunks([self._decode_raw(tokens) for tokens in token_chunks])
        return self._postprocess(
            waveform,
            prompt=prompt,
            config=generation,
        )


__all__ = [
    "OmniVoiceGenerationConfig",
    "OmniVoiceGenerator",
    "OmniVoicePrompt",
]
